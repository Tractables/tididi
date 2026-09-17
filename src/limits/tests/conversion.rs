use std::sync::Arc;

use crate::diagram::{Arithmetic, LiteralWeights, RationalWeights, TddBuildError, WeightStore};
use crate::vtree::{VarId, Vtree};
use crate::{OperationError, Tdd};

/// A function reporting operation errors can attach weights with `?`.
fn weigh(mut f: Tdd, store: WeightStore) -> Result<Tdd, OperationError> {
    f.set_weights(store)?;
    Ok(f)
}

#[test]
fn builder_errors_convert_at_the_operation_boundary() {
    let vtree = Arc::new(Vtree::balanced(2));
    let one = num_rational::BigRational::from_integer(1.into());
    let weights = [LiteralWeights { negative: one.clone(), positive: one }];
    let store = WeightStore::new(RationalWeights::from_literals(&weights), Arithmetic::ExactRational);
    assert_eq!(
        weigh(Tdd::one(&vtree), store).unwrap_err(),
        OperationError::InvalidDiagram(TddBuildError::MissingVariableWeight(VarId(2)))
    );
    assert_eq!(OperationError::from(TddBuildError::IncompatibleWeights), OperationError::IncompatibleWeights);
}
