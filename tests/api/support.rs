use std::sync::Arc;

use num_rational::BigRational;
use num_traits::{One, Zero};
use tididi::diagram::{LiteralWeights, RationalWeights};
pub(super) use tididi::test_helpers::{or_of_cubes, rat, weighted_sum};

use tididi::vtree::VarId;
use tididi::{Engine, Tdd, Vtree};

/// Three shapes/orders over the same dense variable IDs.
pub(super) fn trees(n: u32) -> [Arc<Vtree>; 3] {
    [
        Arc::new(Vtree::balanced(n)),
        Arc::new(Vtree::linear(n)),
        Arc::new(Vtree::linear_from_order(
            &(1..=n).rev().map(VarId).collect::<Vec<_>>(),
        ).unwrap()),
    ]
}

/// Read a variable in an explicitly enumerated assignment.
pub(super) fn bit(row: usize, var: usize) -> bool {
    row & (1 << var) != 0
}

/// Independent Bernoulli weights in variable-ID order.
pub(super) fn bernoulli(probabilities: &[BigRational]) -> Vec<LiteralWeights<BigRational>> {
    probabilities
        .iter()
        .map(|p| LiteralWeights {
            negative: BigRational::one() - p,
            positive: p.clone(),
        })
        .collect()
}

/// Check each assignment, the count over covered variables and a complete witness.
pub(super) fn assert_truth(engine: &Engine, diagram: &Tdd, expected: &[bool], context: &str) {
    let n = diagram.vtree().num_vars();
    assert_eq!(expected.len(), 1 << n);
    for (row, &yes) in expected.iter().enumerate() {
        let weights = bernoulli(
            &(0..n)
                .map(|v| rat(i64::from(bit(row, v as usize)), 1))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            diagram.evaluate(&RationalWeights::from_literals(&weights)).unwrap(),
            rat(i64::from(yes), 1),
            "{context}, assignment {row}"
        );
    }
    assert_eq!(
        engine.model_count(diagram).unwrap(),
        (expected.iter().filter(|&&yes| yes).count() >> (n - diagram.vtree().num_leaves())).into(),
        "{context}, count"
    );
    match engine.satisfying_assignment(diagram).unwrap() {
        None => assert!(
            !expected.iter().any(|&yes| yes),
            "{context}, missing witness"
        ),
        Some(witness) => {
            let mut covered: Vec<_> = diagram.vtree().leaf_bottomup().map(|(_, var)| var).collect();
            covered.sort_unstable();
            assert_eq!(witness.len(), covered.len(), "{context}, complete witness");
            let mut row = 0;
            for (var, literal) in covered.iter().zip(&witness) {
                assert_eq!(
                    literal.var,
                    *var,
                    "{context}, witness variable"
                );
                if literal.sign {
                    row |= 1 << var.idx();
                }
            }
            assert!(expected[row], "{context}, invalid witness {row}");
        }
    }
}

/// Normalize a pair of masses, leaving zero-mass evidence undefined.
pub(super) fn normalized(joint: BigRational, evidence: BigRational) -> Option<BigRational> {
    if evidence.is_zero() {
        None
    } else {
        Some(joint / evidence)
    }
}
