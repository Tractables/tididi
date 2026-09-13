use std::sync::Arc;

use num_rational::BigRational;
use num_traits::{One, Zero};
use tididi::diagram::{LiteralWeights, RationalWeights};
use tididi::query::evaluate;
use tididi::vtree::VarId;
use tididi::{Engine, Literal, Tdd, Vtree};

/// Three shapes/orders over the same dense variable IDs.
pub(super) fn trees(n: u32) -> [Arc<Vtree>; 3] {
    [
        Arc::new(Vtree::balanced(n)),
        Arc::new(Vtree::linear(n)),
        Arc::new(Vtree::linear_from_order(
            &(0..n).rev().map(VarId).collect::<Vec<_>>(),
        )),
    ]
}

/// Read a variable in an explicitly enumerated assignment.
pub(super) fn bit(row: usize, var: usize) -> bool {
    row & (1 << var) != 0
}

/// An exact rational fixture value.
pub(super) fn fraction(n: i64, d: i64) -> BigRational {
    BigRational::new(n.into(), d.into())
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

/// Compile disjoint satisfying cubes on the selected variables; others stay free.
pub(super) fn compile(
    engine: &Engine,
    tree: &Arc<Vtree>,
    vars: &[VarId],
    truth: impl Fn(usize) -> bool,
) -> Tdd {
    let mut result = Tdd::zero(tree);
    for row in 0..1 << vars.len() {
        if truth(row) {
            let cube = engine
                .cube(
                    tree,
                    vars.iter()
                        .enumerate()
                        .map(|(i, &var)| Literal::new(var, bit(row, i))),
                )
                .unwrap();
            result = engine.or(result, cube).unwrap();
        }
    }
    result
}

/// Sum assignment products directly, without consulting a diagram.
pub(super) fn mass(truth: &[bool], weights: &[LiteralWeights<BigRational>]) -> BigRational {
    assert_eq!(truth.len(), 1 << weights.len());
    truth
        .iter()
        .enumerate()
        .filter(|(_, yes)| **yes)
        .map(|(row, _)| {
            weights
                .iter()
                .enumerate()
                .map(|(var, w)| {
                    if bit(row, var) {
                        w.positive.clone()
                    } else {
                        w.negative.clone()
                    }
                })
                .product::<BigRational>()
        })
        .sum()
}

/// Check each assignment, the full-universe count and a total witness via public queries.
pub(super) fn assert_truth(engine: &Engine, diagram: &Tdd, expected: &[bool], context: &str) {
    let n = diagram.vtree().num_vars();
    assert_eq!(expected.len(), 1 << n);
    for (row, &yes) in expected.iter().enumerate() {
        let weights = bernoulli(
            &(0..n)
                .map(|v| fraction(i64::from(bit(row, v as usize)), 1))
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            evaluate(diagram, &RationalWeights::from_literals(&weights)),
            fraction(i64::from(yes), 1),
            "{context}, assignment {row}"
        );
    }
    assert_eq!(
        engine.model_count(diagram).unwrap(),
        expected.iter().filter(|&&yes| yes).count().into(),
        "{context}, count"
    );
    match engine.satisfying_assignment(diagram).unwrap() {
        None => assert!(
            !expected.iter().any(|&yes| yes),
            "{context}, missing witness"
        ),
        Some(witness) => {
            assert_eq!(witness.len(), n as usize, "{context}, total witness");
            let mut row = 0;
            for (var, literal) in witness.iter().enumerate() {
                assert_eq!(
                    literal.var,
                    VarId(var as u32),
                    "{context}, witness variable"
                );
                if literal.positive {
                    row |= 1 << var;
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
