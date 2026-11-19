use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, TddBuildError, Weights};
use crate::vtree::{VarId, Vtree};
use crate::OperationError;

/// A function reporting operation errors can weigh a vtree with `?`.
fn weigh(vtree: Vtree, weights: Weights) -> Result<Vtree, OperationError> {
    Ok(vtree.with_weights(weights)?)
}

#[test]
fn builder_errors_convert_at_the_operation_boundary() {
    let one = num_rational::BigRational::from_integer(1.into());
    let weights = [LiteralWeights { negative: one.clone(), positive: one }];
    let weights = Weights::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational);
    assert_eq!(
        weigh(Vtree::balanced(2), weights).unwrap_err(),
        OperationError::InvalidDiagram(TddBuildError::MissingVariableWeight(VarId(2)))
    );
    assert_eq!(OperationError::from(TddBuildError::IncompatibleWeights), OperationError::IncompatibleWeights);
}
