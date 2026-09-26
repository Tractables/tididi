//! Change a diagram's vtree while preserving its function.
//!
//! [`Tdd::rotation_search`](crate::Tdd::rotation_search) searches for rotations
//! that improve a [`search::RotationObjective`], and
//! [`Tdd::rotate_if`](crate::Tdd::rotate_if) applies rotations one
//! sequence at a time under the caller's own decision.
//! [`Tdd::graft`](crate::Tdd::graft) joins diagrams over disjoint variables on
//! a grafted vtree. [`Tdd::embed`](crate::Tdd::embed) copies one diagram onto a
//! larger vtree under a renaming of its variables.

pub(crate) mod relevel;
pub(crate) mod scratch;
pub mod search;
pub(crate) mod embed;
pub(crate) mod graft;
mod placement;
mod splice;
pub(crate) mod target;

pub use crate::vtree::graft::Embedding;
pub use embed::EmbeddingPlan;
pub use target::RestructureStats;

use crate::diagram::TddBuildError;
use crate::limits::OperationError;
use crate::vtree::{VarId, VtreeError, VtreeIdx};

/// Why a diagram could not be moved onto the requested vtree.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RestructureError {
    /// The two vtrees do not have the same variables: this one is a leaf of
    /// one of them only.
    Variables {
        /// The variable only one vtree has.
        variable: VarId,
    },
    /// The diagram has a weight table, which the move does not carry.
    Weighted,
    /// A rotation at this vtree node would have expanded more pairs than the
    /// bound allows.
    Bound {
        /// The node the refused rotation turns.
        pivot: VtreeIdx,
    },
    /// An operation the move runs was refused: a rotation that would turn a
    /// summed-out level, a refused allocation, or an armed stop.
    Operation(OperationError),
}

impl std::fmt::Display for RestructureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Variables { variable } => write!(f, "variable {} is a leaf of only one of the two vtrees", variable.0),
            Self::Weighted => write!(f, "a weighted diagram cannot be moved to another vtree"),
            Self::Bound { pivot } => write!(f, "the rotation at vtree node {} exceeds the pair bound", pivot.idx()),
            Self::Operation(error) => write!(f, "moving the diagram: {error}"),
        }
    }
}

impl std::error::Error for RestructureError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Operation(error) => Some(error),
            Self::Variables { .. } | Self::Weighted | Self::Bound { .. } => None,
        }
    }
}

impl From<OperationError> for RestructureError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}

/// Why a diagram could not be copied onto the requested vtree.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum EmbedError {
    /// The renamed variables cannot form a vtree: two of them share an image.
    Vtree(VtreeError),
    /// A renamed variable is not a leaf of the destination.
    VariableOutOfRange {
        /// The destination variable.
        variable: VarId,
        /// The largest variable id the destination's id space holds.
        num_vars: u32,
    },
    /// The destination vtree does not contain the source vtree's shape under
    /// the renaming, so no level-by-level copy exists.
    NotIsomorphic {
        /// The source vtree node the destination stopped matching at.
        source: VtreeIdx,
    },
    /// An operation the copy runs was refused: a source that has discarded
    /// the structure at a level, a refused allocation, or an armed stop.
    Operation(OperationError),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vtree(error) => write!(f, "embedding: {error}"),
            Self::VariableOutOfRange { variable, num_vars } => write!(f, "renamed variable {} is outside the variables 1 to {num_vars}", variable.0),
            Self::NotIsomorphic { source } => write!(f, "the destination vtree does not contain the source vtree's shape at node {}", source.idx()),
            Self::Operation(error) => write!(f, "copying the diagram: {error}"),
        }
    }
}

impl std::error::Error for EmbedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vtree(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::VariableOutOfRange { .. } | Self::NotIsomorphic { .. } => None,
        }
    }
}

impl From<VtreeError> for EmbedError {
    fn from(error: VtreeError) -> Self { Self::Vtree(error) }
}

impl From<OperationError> for EmbedError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}

/// Why diagrams could not be joined on a grafted vtree with their structure
/// and weight interpretation preserved.
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
        /// The largest variable id the destination's id space holds.
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
    /// An operation the assembly runs was refused: an operand that has
    /// discarded the structure at a level, a refused allocation, or an armed
    /// stop.
    Operation(OperationError),
}

impl std::fmt::Display for GraftError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vtree(error) => write!(f, "grafted vtree: {error}"),
            Self::MissingVariableMapping { part, variable } => write!(f, "part {part} has no mapping for variable {}", variable.0),
            Self::VariableOutOfRange { variable, num_vars } => write!(f, "grafted variable {} is outside the variables 1 to {num_vars}", variable.0),
            Self::PartWeights { part, source } => write!(f, "part {part}: {source}"),
            Self::DestinationWeights(error) => write!(f, "graft destination: {error}"),
            Self::Operation(error) => write!(f, "assembling the result: {error}"),
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
