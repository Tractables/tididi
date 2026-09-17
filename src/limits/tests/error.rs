//! What the error surface promises a consumer that matches on it.

use crate::OperationError;
use std::error::Error;

/// `IndexOverflow` is not a resource condition, and the two must not print or
/// compare as one: a consumer that retries `OverBudget` with a larger budget
/// would loop forever on the other.
#[test]
fn an_index_overflow_is_not_an_over_budget() {
    assert_ne!(OperationError::IndexOverflow, OperationError::OverBudget);
    assert!(!OperationError::IndexOverflow.to_string().contains("budget"));
}

/// `InvalidDiagram` reaches its cause through `source()`, so `{:#}`-style
/// chain printing and `anyhow`-style walkers see the storage check.
#[test]
fn an_invalid_diagram_exposes_the_storage_check_it_wraps() {
    let inner = crate::diagram::TddBuildError::LevelCountMismatch { expected: 3, found: 2 };
    let err = OperationError::InvalidDiagram(inner);
    let source = err.source().expect("the wrapped storage check");
    assert_eq!(source.to_string(), inner.to_string());
    assert!(OperationError::OverBudget.source().is_none());
}
