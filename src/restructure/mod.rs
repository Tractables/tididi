//! Change a diagram's vtree while preserving its function.
//!
//! [`Tdd::rotation_search`](crate::Tdd::rotation_search) searches for rotations
//! that improve a [`search::RotationObjective`]. [`Tdd::graft`](crate::Tdd::graft)
//! joins diagrams over disjoint variables on a grafted vtree.

pub(crate) mod relevel;
pub(crate) mod scratch;
pub mod search;
pub(crate) mod graft;

use crate::diagram::TddBuildError;
use crate::limits::OperationError;
use crate::vtree::{VarId, VtreeError};

/// Why a diagram graft could not preserve its structure and weight interpretation.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum GraftError {
    /// The renamed variable sets cannot form a vtree.
    Vtree(VtreeError),
    /// A part's map has no entry for one of its variables.
    MissingVariableMapping {
        /// The part's position in the input vector.
        part: usize,
        /// The unmapped local variable.
        variable: VarId,
    },
    /// A renamed or free variable is outside the destination's id space.
    VariableOutOfRange {
        /// The destination variable.
        variable: VarId,
        /// The exclusive upper bound on variable ids.
        num_vars: u32,
    },
    /// A part's stored values cannot be interpreted in the requested destination.
    PartWeights {
        /// The part's position in the input vector.
        part: usize,
        /// The inconsistent weight or marginal-level state.
        source: TddBuildError,
    },
    /// The destination weight table does not cover the result's variables or columns.
    DestinationWeights(TddBuildError),
    /// Cleanup of a newly attached marginal root was refused by the engine.
    Operation(OperationError),
}

impl std::fmt::Display for GraftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vtree(error) => write!(f, "grafted vtree: {error}"),
            Self::MissingVariableMapping { part, variable } => write!(f, "part {part} has no mapping for variable {}", variable.idx()),
            Self::VariableOutOfRange { variable, num_vars } => write!(f, "grafted variable {} is outside the id space 0..{num_vars}", variable.idx()),
            Self::PartWeights { part, source } => write!(f, "part {part}: {source}"),
            Self::DestinationWeights(error) => write!(f, "graft destination: {error}"),
            Self::Operation(error) => write!(f, "graft cleanup: {error}"),
        }
    }
}

impl std::error::Error for GraftError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vtree(error) => Some(error),
            Self::PartWeights { source, .. } | Self::DestinationWeights(source) => Some(source),
            Self::Operation(error) => Some(error),
            Self::MissingVariableMapping { .. } | Self::VariableOutOfRange { .. } => None,
        }
    }
}

impl From<VtreeError> for GraftError {
    fn from(error: VtreeError) -> Self { Self::Vtree(error) }
}

impl From<OperationError> for GraftError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}
