//! Change a diagram's vtree while preserving its function.
//!
//! [`Tdd::rotation_search`](crate::Tdd::rotation_search) searches for rotations
//! that improve a [`search::RotationObjective`], and
//! [`Tdd::rotate_if`](crate::Tdd::rotate_if) applies rotations one
//! sequence at a time under the caller's own decision.
//! [`Tdd::graft`](crate::Tdd::graft) joins diagrams over disjoint variables on
//! a grafted vtree. [`Tdd::embed`](crate::Tdd::embed) copies one diagram onto a
//! larger vtree under a renaming of its variables, and
//! [`Tdd::expand_variables`](crate::Tdd::expand_variables) expands each of its
//! variables into a class of new ones.

pub(crate) mod relevel;
pub(crate) mod scratch;
pub mod search;
pub(crate) mod embed;
pub(crate) mod expand;
pub(crate) mod graft;
mod distinct;
pub(crate) mod placement;
mod project;
mod tag;
mod splice;
pub(crate) mod target;

pub use crate::vtree::graft::Embedding;
pub use embed::EmbeddingPlan;
pub use expand::VariableExpansion;
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
    /// The diagram's stored values cannot be interpreted in the destination:
    /// a weight-marginal level without a destination store, or a destination
    /// table that disagrees with the diagram's under the renaming.
    SourceWeights(TddBuildError),
    /// The destination weight table does not cover the result's variables or
    /// columns.
    DestinationWeights(TddBuildError),
    /// An operation the copy runs was refused: a source that has discarded
    /// the structure at a level, a refused allocation, or an armed stop.
    Operation(OperationError),
}

/// An embedding [`Engine::embed_moving`](crate::Engine::embed_moving)
/// refused, with the diagram as it was given.
#[derive(Debug)]
pub struct EmbedRefused {
    /// Why the embedding was refused.
    pub error: EmbedError,
    /// The diagram, unchanged.
    pub tdd: crate::Tdd,
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vtree(error) => write!(f, "embedding: {error}"),
            Self::VariableOutOfRange { variable, num_vars } => write!(f, "renamed variable {} is outside the variables 1 to {num_vars}", variable.0),
            Self::NotIsomorphic { source } => write!(f, "the destination vtree does not contain the source vtree's shape at node {}", source.idx()),
            Self::SourceWeights(error) => write!(f, "embedded diagram: {error}"),
            Self::DestinationWeights(error) => write!(f, "embedding destination: {error}"),
            Self::Operation(error) => write!(f, "copying the diagram: {error}"),
        }
    }
}

impl std::error::Error for EmbedError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vtree(error) => Some(error),
            Self::SourceWeights(error) | Self::DestinationWeights(error) => Some(error),
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

/// Why a diagram could not be expanded as [`Tdd::expand_variables`](crate::Tdd::expand_variables) asks.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ExpandError {
    /// The expanded variables cannot form a vtree: one is named twice, or
    /// their id space is too wide for the result.
    Vtree(VtreeError),
    /// A variable of the diagram's vtree has no class, or an empty one.
    MissingClass {
        /// The variable without a class.
        variable: VarId,
    },
    /// A variable the diagram's vtree does not carry has a nonempty class.
    ClassWithoutLeaf {
        /// The variable the class was given for.
        variable: VarId,
    },
    /// An expanded variable is outside the expansion's id space.
    VariableOutOfRange {
        /// The expanded variable.
        variable: VarId,
        /// The largest variable id the id space holds.
        num_vars: u32,
    },
    /// An operation the expansion runs was refused: a diagram that has summed
    /// out a level, a refused allocation, or an armed stop.
    Operation(OperationError),
}

impl std::fmt::Display for ExpandError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vtree(error) => write!(f, "expanded vtree: {error}"),
            Self::MissingClass { variable } => write!(f, "variable {} of the diagram has no class", variable.0),
            Self::ClassWithoutLeaf { variable } => write!(f, "variable {} has a class but is not a leaf of the diagram's vtree", variable.0),
            Self::VariableOutOfRange { variable, num_vars } => write!(f, "expanded variable {} is outside the variables 1 to {num_vars}", variable.0),
            Self::Operation(error) => write!(f, "expanding the diagram: {error}"),
        }
    }
}

impl std::error::Error for ExpandError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Vtree(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::MissingClass { .. } | Self::ClassWithoutLeaf { .. } | Self::VariableOutOfRange { .. } => None,
        }
    }
}

impl From<VtreeError> for ExpandError {
    fn from(error: VtreeError) -> Self { Self::Vtree(error) }
}

impl From<OperationError> for ExpandError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}

/// Why the nodes of a level could not be tagged with codes
/// ([`Engine::tag_level`](crate::Engine::tag_level)).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TagError {
    /// The destination's root has no left subtree holding the level's
    /// subtree under the renaming, as an embedding reports it.
    Placement(EmbedError),
    /// The code variables are not exactly the leaves of the destination
    /// root's right subtree.
    CodeVariables(VtreeError),
    /// The tags or the codes do not fit the level: a tag count other than
    /// the level's width, a tag naming no code, a code list that is not a
    /// whole number of codes, or overlapping leaf slots tagged.
    Tags(String),
    /// An operation the construction runs was refused: a diagram that has
    /// discarded the structure at a level, a level that is not in the
    /// vtree, a refused allocation, or an armed stop.
    Operation(OperationError),
}

impl std::fmt::Display for TagError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Placement(error) => write!(f, "tagged level: {error}"),
            Self::CodeVariables(error) => write!(f, "code variables: {error}"),
            Self::Tags(message) => write!(f, "tags: {message}"),
            Self::Operation(error) => write!(f, "tagging the level: {error}"),
        }
    }
}

impl std::error::Error for TagError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Placement(error) => Some(error),
            Self::CodeVariables(error) => Some(error),
            Self::Operation(error) => Some(error),
            Self::Tags(_) => None,
        }
    }
}

impl From<OperationError> for TagError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}

/// Why the distinct values under a subtree could not be counted
/// ([`Engine::at_least_distinct`](crate::Engine::at_least_distinct)).
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DistinctError {
    /// The key subtree is the root, or its parent is neither the root nor a
    /// child of the root, so the count is not a sum over one level's pairs.
    Placement {
        /// The key subtree's root.
        key: VtreeIdx,
    },
    /// An operation the pass runs was refused: a key that is not in the
    /// vtree, a diagram that has discarded the structure at a level, a
    /// refused allocation, or an armed stop.
    Operation(OperationError),
}

impl std::fmt::Display for DistinctError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Placement { key } => {
                write!(f, "vtree node {} is not a child or a grandchild of the root", key.idx())
            }
            Self::Operation(error) => write!(f, "counting distinct values: {error}"),
        }
    }
}

impl std::error::Error for DistinctError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Operation(error) => Some(error),
            Self::Placement { .. } => None,
        }
    }
}

impl From<OperationError> for DistinctError {
    fn from(error: OperationError) -> Self { Self::Operation(error) }
}
